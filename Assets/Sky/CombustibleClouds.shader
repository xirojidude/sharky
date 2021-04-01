
Shader "Skybox/CombustibleClouds"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }


/*

    Combustible Clouds
    ------------------
    
    This is just a daytime version of my cheap cloud flythrough example. I'm not sure why
    the clouds exist in a combustible atmosphere, or if that's even possible... but it's 
    just a cheap hack, so isn't meant to be taken seriously. :)

    Obviously, the object of the exercise was speed, rather than quality, so it should run
    pretty well on even the slowest of machines. With that said, the quality is pretty good,
    all things considered.
    
    Based on:
    
    Cloudy Spikeball - Duke
    https://www.shadertoy.com/view/MljXDw
    // Port from a demo by Las - Worth watching.
    // http://www.pouet.net/topic.php?which=7920&page=29&x=14&y=9

*/

// Hash function. This particular one probably doesn't disperse things quite 
// as nicely as some of the others around, but it's compact, and seems to work.
//
float3 hash33(float3 p){ 
    float n = sin(dot(p, float3(7, 157, 113)));    
    return frac(float3(2097152, 262144, 32768)*n); 
}


// IQ's tex2D lookup noise... in obfuscated form. There's less writing, so
// that makes it faster. That's how optimization works, right? :) Seriously,
// though, refer to IQ's original for the proper function.
// 
// By the way, you could replace this with the non-tex2Dd version, and the
// shader should run at almost the same efficiency.
float n3D( in float3 p ){
    
    float3 i = floor(p); p -= i; p *= p*(3. - 2.*p);
    p.xy = tex2D(_MainTex, (p.xy + i.xy + float2(37, 17)*i.z + .5)/256.).yx;
    return lerp(p.x, p.y, p.z);
}

/*
// tex2Dless 3D Value Noise:
//
// This is a rewrite of IQ's original. It's self contained, which makes it much
// easier to copy and paste. I've also tried my best to minimize the amount of 
// operations to lessen the work the GPU has to do, but I think there's room for
// improvement. I have no idea whether it's faster or not. It could be slower,
// for all I know, but it doesn't really matter, because in its current state, 
// it's still no match for IQ's tex2D-based, smooth 3D value noise.
//
// By the way, a few people have managed to reduce the original down to this state, 
// but I haven't come across any who have taken it further. If you know of any, I'd
// love to hear about it.
//
// I've tried to come up with some clever way to improve the randomization line
// (h = lerp(frac...), but so far, nothing's come to mind.
float n3D(float3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
    const float3 s = float3(7, 157, 113);
    
    float3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride floattor for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    float4 h = float4(0., s.yz, s.y + s.z) + dot(ip, s);
    
    p -= ip; // Cell's fracional component.
    
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Smoother version of the above. Weirdly, the extra calculations can sometimes
    // create a surface that's easier to hone in on, and can actually speed things up.
    // Having said that, I'm sticking with the simpler version above.
    //p = p*p*p*(p*(p * 6. - 15.) + 10.);
    
    // Even smoother, but this would have to be slower, surely?
    //float3 p3 = p*p*p; p = ( 7. + ( p3 - 7. ) * p ) * p3;   
    
    // Cosinusoidal smoothing. OK, but I prefer other methods.
    //p = .5 - .5*cos(p*3.14159);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: frac(sin(x)*largeNumber).
    h = lerp(frac(sin(h)*43758.5453), frac(sin(h + s.x)*43758.5453), p.x);
    
    // Interpolating along Y.
    h.xy = lerp(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return lerp(h.x, h.y, p.z); // Range: [0, 1].
    
}

*/

// Basic low quality noise consisting of three layers of rotated, mutated 
// trigonometric functions. Needs work, but sufficient for this example.
float trigNoise3D(in float3 p){

    
    float res = 0., sum = 0.;
    
    // IQ's cheap, tex2D-lookup noise function. Very efficient, but still 
    // a little too processor intensive for multiple layer usage in a largish 
    // "for loop" setup. Therefore, just one layer is being used here.
    float n = n3D(p*8. + _Time.y*2.);


    // Two sinusoidal layers. I'm pretty sure you could get rid of one of 
    // the swizzles (I have a feeling the GPU doesn't like them as much), 
    // which I'll try to do later.
    
    float3 t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    p = p*1.5 + (t - 1.5); //  + _Time.y*0.1
    res += (dot(t, float3(0.333,0.333,0.333)));

    t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    res += (dot(t, float3(0.333,0.333,0.333)))*0.7071;    
     
    return ((res/1.7071))*0.85 + n*0.15;
}

// Distance function.
float map(float3 p) {

    return trigNoise3D(p*.5);
    
    // Three layers of noise, for comparison.
    //p += _Time.y;
    //return n3D(p*.75)*.57 + n3D(p*1.875)*.28 + n3D(p*4.6875)*.15;
}



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin


    // Unit direction ray floattor: Note the absence of a divide term. I came across
    // this via a comment Shadertoy user "coyote" made. I'm pretty easy to please,
    // but I thought it was pretty cool.
//    float3 rd = normalize(float3(fragCoord - iResolution.xy*.5, iResolution.y*.75)); 

    // Ray origin. Moving along the Z-axis.
//    float3 ro = float3(0, 0, _Time.y*4.);

    // Cheap camera rotation.
    //
    // 2D rotation matrix. Note the absence of a cos variable. It's there, but in disguise.
    // This one came courtesy of Shadertoy user, "Fabrice Neyret."
    float2 a = sin(float2(1.5707963, 0) + _Time.y*0.1875); 
    float2x2 rM = float2x2(a, -a.y, a.x);
    
    rd.xy = mul(rd.xy,rM); // Apparently, "rd.xy *= rM" doesn't work on some setups. Crazy.
    a = sin(float2(1.5707963, 0) + cos(_Time.y*0.1875*.7)*.7);
    rM = float2x2(a, -a.y, a.x); 
    rd.xz = mul(rd.xz,rM);

    // Placing a light in front of the viewer and up a little. You could just make the 
    // light directional and be done with it, but giving it some point-like qualities 
    // makes it a little more interesting. You could also rotate it in sync with the 
    // camera, like a light beam from a flying vehicle.
    float3 lp = float3(0, 1, 6);
    //lp.xz = lp.xz*rM;
    lp += ro;
    
    

    // The ray is effectively marching through discontinuous slices of noise, so at certain
    // angles, you can see the separation. A bit of randomization can mask that, to a degree.
    // At the end of the day, it's not a perfect process. Note, the ray is deliberately left 
    // unnormalized... if that's a word.
    //
    // Randomizing the direction.
    //rd = (rd + (hash33(rd.zyx)*0.004-0.002)); 
    // Randomizing the length also. 
    //rd *= (1. + frac(sin(dot(float3(7, 157, 113), rd.zyx))*43758.5453)*0.04-0.02);  
    
    //rd = rd*.5 + normalize(rd)*.5;    
    
    // Some more randomization, to be used for color based jittering inside the loop.
    float3 rnd = hash33(rd + 311.);

    // Local density, total density, and weighting factor.
    float lDe = 0., td = 0., w = 0.;

    // Closest surface distance, and total ray distance travelled.
    float d = 1., t = dot(rnd, float3(.08,.08,.08));

    // Distance threshold. Higher numbers give thicker clouds, but fill up the screen too much.    
    const float h = .5;


    // Initializing the scene color to black, and declaring the surface position floattor.
    float3 col = float3(0,0,0), sp;



    // Particle surface normal.
    //
    // Here's my hacky reasoning. I'd imagine you're going to hit the particle front on, so the normal
    // would just be the opposite of the unit direction ray. However particles are particles, so there'd
    // be some randomness attached... Yeah, I'm not buying it either. :)
    float3 sn = normalize(hash33(rd.yxz)*.03-rd);

    // Raymarching loop.
    for (int i=0; i<64; i++) {

        // Loop break conditions. Seems to work, but let me
        // know if I've overlooked something.
        if((td>1.) || d<.001*t || t>80.)break;


        sp = ro + rd*t; // Current ray position.
        d = map(sp); // Closest distance to the surface... particle.

        // If we get within a certain distance, "h," of the surface, accumulate some surface values.
        // The "step" function is a branchless way to do an if statement, in case you're wondering.
        //
        // Values further away have less influence on the total. When you accumulate layers, you'll
        // usually need some kind of weighting algorithm based on some identifying factor - in this
        // case, it's distance. This is one of many ways to do it. In fact, you'll see variations on 
        // the following lines all over the place.
        //
        lDe = (h - d) * step(d, h); 
        w = (1. - td) * lDe;   

        // Use the weighting factor to accumulate density. How you do this is up to you. 
        td += w*w*8. + 1./60.; //w*w*5. + 1./50.;
        //td += w*.4 + 1./45.; // Looks cleaner, but a little washed out.


        // Point light calculations.
        float3 ld = lp-sp; // Direction floattor from the surface to the light position.
        float lDist = max(length(ld), 0.001); // Distance from the surface to the light.
        ld/=lDist; // Normalizing the directional light floattor.

        // Using the light distance to perform some falloff.
        float atten = 1./(1. + lDist*0.1 + lDist*lDist*.03);

        // Ok, these don't entirely correlate with tracing through transparent particles,
        // but they add a little anglular based highlighting in order to fake proper lighting...
        // if that makes any sense. I wouldn't be surprised if the specular term isn't needed,
        // or could be taken outside the loop.
        float diff = max(dot(sn, ld ), 0.);
        float spec = pow(max(dot( reflect(-ld, sn), -rd ), 0.), 4.);


        // Accumulating the color. Note that I'm only adding a scalar value, in this case,
        // but you can add color combinations.
        col += w*(1.+ diff*.5 + spec*.5)*atten;
        // Optional extra: Color-based jittering. Roughens up the grey clouds that hit the camera lens.
        col += (frac(rnd*289. + t*41.) - .5)*.02;;

        // Try this instead, to see what it looks like without the fake contrasting. Obviously,
        // much faster.
        //col += w*atten*1.25;


        // Enforce minimum stepsize. This is probably the most important part of the procedure.
        // It reminds me a little of the soft shadows routine.
        t +=  max(d*.5, .02); //
        // t += 0.2; // t += d*0.5;// These also work, but don't seem as efficient.

    }
    
    col = max(col, 0.);

    
    // Adding a bit of a firey tinge to the cloud value.
    col = lerp(pow(float3(1.3, 1, 1)*col, float3(1, 2, 10)), col, dot(cos(rd*6. +sin(rd.yzx*6.)), float3(.333,.333,.333))*.2+.8);
 
    // Using the light position to produce a blueish sky and sun. Pretty standard.
    float3 sky = float3(.6, .8, 1.)*min((1.5+rd.y*.5)/2., 1.);  
    sky = lerp(float3(1, 1, .9), float3(.31, .42, .53), rd.y*0.5 + 0.5);
    
    float sun = clamp(dot(normalize(lp-ro), rd), 0.0, 1.0);
   
    // Combining the clouds, sky and sun to produce the final color.
    sky += float3(1, .3, .05)*pow(sun, 5.)*.25; 
    sky += float3(1, .4, .05)*pow(sun, 16.)*.35;  
    col = lerp(col, sky, smoothstep(0., 25., t));
    col += float3(1, .6, .05)*pow(sun, 16.)*.25;  
 
    // Done.
    fragColor = float4(sqrt(min(col, 1.)), 1.0);
 

                return fragColor;
            }

            ENDCG
        }
    }
}


//https://www.shadertoy.com/view/MscXRH
