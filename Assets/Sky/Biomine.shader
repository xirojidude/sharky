
Shader "Skybox/Biomine"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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
    Biomine
    -------
    
    A biocooling system for a futuristic, off-world mine... or a feeding mechanisn for an alien 
    hatchery? I wasn't really sure what I was creating when I started, and I'm still not. :) I at 
    least wanted to create the sense that the tubes were pumping some form of biomatter around 
    without having to resort to full reflective and refracive passes... I kind of got there. :)

    All things considered, there's not a lot to this. Combine a couple of gyroid surfaces, ID them,
    then add their respective material properties. The scene is simple to create, and explained in
    the distance function. There's also some function based, 2nd order cellular bump mapping, for 
    anyone interested.

    The fluid pumped through the tubes was created by indexing the reflected and refraced rays 
    into a basic environment mapping function. Not accurate, but simple, effective and way cheaper
    than the real thing.

    I'd just finished watching some of the Assembly 2016 entries on YouTube, so for better or
    worse, wanted to produce the scene without the help of any in-house textures.

    Related examples: 

    Cellular Tiling - Shane
    https://www.shadertoy.com/view/4scXz2

    Cellular Tiled Tunnel - Shane
    https://www.shadertoy.com/view/MscSDB

*/

// Max ray distance.
#define FAR 50. 

// Variables used to identify the objects. In this case, there are just two - the biotubes and
// the tunnel walls.
float objID = 0.; // Biotubes: 0, Tunnel walls: 1.
float saveID = 0.;


// Standard 1x1 hash functions. Using "cos" for non-zero origin result.
float hash( float n ){ return frac(cos(n)*45758.5453); }



// 2x2 matrix rotation. Note the absence of "cos." It's there, but in disguise, and comes courtesy
// of Fabrice Neyret's "ouside the box" thinking. :)
float2x2 rot2( float a ){ float2 v = sin(float2(1.570796, 0) + a);  return float2x2(v, -v.y, v.x); }


// Compact, self-contained version of IQ's 3D value noise function. I have a transparent noise
// example that explains it, if you require it.
float noise3D(in float3 p){
    
    const float3 s = float3(7, 157, 113);
    float3 ip = floor(p); p -= ip; 
    float4 h = float4(0., s.yz, s.y + s.z) + dot(ip, s);
    p = p*p*(3. - 2.*p); //p *= p*p*(p*(p * 6. - 15.) + 10.);
    h = lerp(frac(sin(h)*43758.5453), frac(sin(h + s.x)*43758.5453), p.x);
    h.xy = lerp(h.xz, h.yw, p.y);
    return lerp(h.x, h.y, p.z); // Range: [0, 1].
}

////////
// The cellular tile routine. Draw a few objects (four spheres, in this case) using a minumum
// blend at various 3D locations on a cubic tile. Make the tile wrappable by ensuring the 
// objects wrap around the edges. That's it.
//
// Believe it or not, you can get away with as few as three spheres. If you sum the total 
// instruction count here, you'll see that it's way, way lower than 2nd order 3D Voronoi.
// Not requiring a hash function provides the biggest benefit, but there is also less setup.
// 
// The result isn't perfect, but 3D cellular tiles can enable you to put a Voronoi looking 
// surface layer on a lot of 3D objects for little cost.
//
float drawSphere(in float3 p){
  
    p = frac(p)-.5;    
    return dot(p, p);
    
    //p = abs(frac(p)-.5);
    //return dot(p, float3(.5));  
}


float cellTile(in float3 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    float4 v, d; 
    d.x = drawSphere(p - float3(.81, .62, .53));
    p.xy = float2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawSphere(p - float3(.39, .2, .11));
    p.yz = float2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawSphere(p - float3(.62, .24, .06));
    p.xz = float2(p.z-p.x, p.z + p.x)*.7071;
    d.w = drawSphere(p - float3(.2, .82, .64));

    v.xy = min(d.xz, d.yw), v.z = min(max(d.x, d.y), max(d.z, d.w)), v.w = max(v.x, v.y); 
   
    d.x =  min(v.z, v.w) - min(v.x, v.y); // First minus second order, for that beveled Voronoi look. Range [0, 1].
    //d.x =  min(v.x, v.y); // Minimum, for the cellular look.
        
    return d.x*2.66; // Normalize... roughly.
    
}

// The path is a 2D sinusoid that varies over time, depending upon the frequencies, and amplitudes.
float2 path(in float z){ 
    //return float2(0);
    float a = sin(z * 0.11);
    float b = cos(z * 0.14);
    return float2(a*4. - b*1.5, b*1.7 + a*1.5); 
}


// Smooth maximum, based on IQ's smooth minimum function.
float smaxP(float a, float b, float s){
    
    float h = clamp( 0.5 + 0.5*(a-b)/s, 0., 1.);
    return lerp(b, a, h) + h*(1.0-h)*s;
}


// The distance function. It's a lot simpler than it looks: The biological tubes are just a gyroid lattice.
// The mine tunnel, is created by takoing the negative space, and bore out the center with a cylinder. The
// two are combined with a smooth maximum to produce the tunnel with biotube lattice. On top of that, the 
// whole scene is wrapped around a path and slightly mutated (the first two lines), but that's it.

float map(float3 p){
  
    p.xy -= path(p.z); // Wrap the scene around a path.

    p += cos(p.zxy*1.5707963)*.2; // Perturb slightly. The mutation gives it a bit more of an organic feel.

    // If you're not familiar with a gyroid lattice, this is basically it. Not so great to hone in on, but
    // pretty cool looking and simple to produce.
    float d = dot(cos(p*1.5707963), sin(p.yzx*1.5707963)) + 1.;

    // Biotube lattice. The final time-based term makes is heave in and out.
    float bio = d + .25 +  dot(sin(p*1. + _Time.y*6.283 + sin(p.yzx*.5)), float3(.033,.033,.033));

    // The tunnel. Created with a bit of trial and error. The smooth maximum against the gyroid rounds it off
    // a bit. The abs term at the end just adds some variation via the beveled edges. Also trial and error.
    float tun = smaxP(3.25 - length(p.xy - float2(0, 1)) + .5*cos(p.z*3.14159/32.), .75-d, 1.) - abs(1.5-d)*.375;;// - sf*.25;


    objID = step(tun, bio); // Tunnel and biolattice IDs, for coloring, lighting, bumping, etc, later.

    return min(tun, bio); // Return the distance to the scene.

 
}


// Surface bump function. Cheap, but with decent visual impact.
float bumpSurf3D( in float3 p){
    
    float bmp;
    float noi = noise3D(p*96.);
    
    if(saveID>.5){
        float sf = cellTile(p*.75); 
        float vor = cellTile(p*1.5);
    
        bmp = sf*.66 + (vor*.94 + noi*.06)*.34;
    }
    else {
        p/=3.;//
        float ct = cellTile(p*2. + sin(p*12.)*.5)*.66+cellTile(p*6. + sin(p*36.)*.5)*.34;
        bmp = (1.-smoothstep(-.2, .25, ct))*.9 + noi*.1;

        
    }
    
    return bmp;

}

// Standard function-based bump mapping function.
float3 doBumpMap(in float3 p, in float3 nor, float bumpfactor){
    
    const float2 e = float2(0.001, 0);
    float ref = bumpSurf3D(p);                 
    float3 grad = (float3(bumpSurf3D(p - e.xyy),
                      bumpSurf3D(p - e.yxy),
                      bumpSurf3D(p - e.yyx) )-ref)/e.x;                     
          
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
    
}

// Basic raymarcher.
float trace(in float3 ro, in float3 rd){

    float t = 0.0, h;
    for(int i = 0; i < 72; i++){
    
        h = map(ro+rd*t);
        // Note the "t*b + a" addition. Basically, we're putting less emphasis on accuracy, as
        // "t" increases. It's a cheap trick that works in most situations... Not all, though.
        if(abs(h)<0.002*(t*.125 + 1.) || t>FAR) break; // Alternative: 0.001*max(t*.25, 1.)        
        t += step(h, 1.)*h*.2 + h*.5;
        
    }

    return min(t, FAR);
}

// Standard normal function. It's not as fast as the tetrahedral calculation, but more symmetrical.
float3 getNormal(in float3 p) {
    const float2 e = float2(0.002, 0);
    return normalize(float3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy), map(p + e.yyx) - map(p - e.yyx)));
}

// XT95's really clever, cheap, SSS function. The way I've used it doesn't do it justice,
// so if you'd like to really see it in action, have a look at the following:
//
// Alien Cocoons - XT95: https://www.shadertoy.com/view/MsdGz2
//
float thickness( in float3 p, in float3 n, float maxDist, float falloff )
{
    const float nbIte = 6.0;
    float ao = 0.0;
    
    for( float i=1.; i< nbIte+.5; i++ ){
        
        float l = (i*.75 + frac(cos(i)*45758.5453)*.25)/nbIte*maxDist;
        
        ao += (l + map( p -n*l )) / pow(1. + l, falloff);
    }
    
    return clamp( 1.-ao/nbIte, 0., 1.);
}

/*
// Shadows.
float softShadow(float3 ro, float3 rd, float start, float end, float k){

    float shade = 1.0;
    const int maxIterationsShad = 20;

    float dist = start;
    //float stepDist = end/float(maxIterationsShad);

    // Max shadow iterations - More iterations make nicer shadows, but slow things down.
    for (int i=0; i<maxIterationsShad; i++){
    
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);

        // +=h, +=clamp( h, 0.01, 0.25 ), +=min( h, 0.1 ), +=stepDist, +=min(h, stepDist*2.), etc.
        dist += clamp( h, 0.01, 0.25);//min(h, stepDist);
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0.001 || dist > end) break; 
    }

    // Shadow value.
    return min(max(shade, 0.) + 0.5, 1.0); 
}
*/


// Ambient occlusion, for that self shadowed look. Based on the original by XT95. I love this 
// function, and in many cases, it gives really, really nice results. For a better version, and 
// usage, refer to XT95's examples below:
//
// Hemispherical SDF AO - https://www.shadertoy.com/view/4sdGWN
// Alien Cocoons - https://www.shadertoy.com/view/MsdGz2
float calculateAO( in float3 p, in float3 n )
{
    float ao = 0.0, l;
    const float maxDist = 4.;
    const float nbIte = 6.0;
    //const float falloff = 0.9;
    for( float i=1.; i< nbIte+.5; i++ ){
    
        l = (i + hash(i))*.5/nbIte*maxDist;
        
        ao += (l - map( p + n*l ))/(1.+ l);// / pow(1.+l, falloff);
    }
    
    return clamp(1.- ao/nbIte, 0., 1.);
}

/*
/////
// Code block to produce some layers of smokey haze. Not sophisticated at all.
// If you'd like to see a much more sophisticated version, refer to Nitmitz's
// Xyptonjtroz example. Incidently, I wrote this off the top of my head, but
// I did have that example in mind when writing this.

// Hash to return a scalar value from a 3D floattor.
float hash31(float3 p){ return frac(sin(dot(p, float3(127.1, 311.7, 74.7)))*43758.5453); }

// Four layers of cheap cell tile noise to produce some subtle mist.
// Start at the ray origin, then take four samples of noise between it
// and the surface point. Apply some very simplistic lighting along the 
// way. It's not particularly well thought out, but it doesn't have to be.
float getMist(in float3 ro, in float3 rd, in float3 lp, in float t){

    float mist = 0.;
    ro += rd*t/64.; // Edge the ray a little forward to begin.
    
    for (int i = 0; i<8; i++){
        // Lighting. Technically, a lot of these points would be
        // shadowed, but we're ignoring that.
        float sDi = length(lp-ro)/FAR; 
        float sAtt = min(1./(1. + sDi*0.25 + sDi*sDi*0.25), 1.);
        // Noise layer.
        //float n = trigNoise3D(ro/2.);//noise3D(ro/2.)*.66 + noise3D(ro/1.)*.34;
        float n = cellTile(ro/2.);
        mist += n*sAtt;//trigNoise3D
        // Advance the starting point towards the hit point.
        ro += rd*t/8.;
    }
    
    // Add a little noise, then clamp, and we're done.
    return clamp(mist/4. + hash31(ro)*0.2-0.1, 0., 1.);

}
*/

//////
// Simple environment mapping. Pass the reflected floattor in and create some
// colored noise with it. The normal is redundant here, but it can be used
// to pass into a 3D texture mapping function to produce some interesting
// environmental reflections.
//
// More sophisticated environment mapping:
// UI easy to integrate - XT95    
// https://www.shadertoy.com/view/ldKSDm
float3 eMap(float3 rd, float3 sn){
    
    
    // Add a time component, scale, then pass into the noise function.
    rd.y += _Time.y;
    rd /= 3.;

    // Biotube texturing.
    float ct = cellTile(rd*2. + sin(rd*12.)*.5)*.66 + cellTile(rd*6. + sin(rd*36.)*.5)*.34;
    float3 texCol = (float3(.25, .2, .15)*(1.-smoothstep(-.1, .3, ct)) + float3(0.02, 0.02, 0.53)/6.); 
    return smoothstep(0., 1., texCol);
    
}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.01+ _XYZPos;                                             // ray origin

    
    // Screen coordinates.
//    float2 uv = (fragCoord - iResolution.xy*0.5)/iResolution.y;
    
    // Camera Setup.
    float3 lookAt = float3(0, 1, _Time.y*2. + 0.1);  // "Look At" position.
    float3 camPos = ro ;//+ lookAt + float3(0.0, 0.0, -0.1); // Camera position, doubling as the ray origin.

 
    // Light positioning. 
    float3 lightPos = camPos + float3(0, .5, 5);// Put it a bit in front of the camera.

    // Using the Z-value to perturb the XY-plane.
    // Sending the camera, "look at," and light floattor down the tunnel. The "path" function is 
    // synchronized with the distance function.
//    lookAt.xy += path(lookAt.z);
//    camPos.xy += path(camPos.z);
//    lightPos.xy += path(lightPos.z);

    // Using the above to produce the unit ray-direction floattor.
    float FOV = 3.14159265/2.; // FOV - Field of view.
    float3 forward = normalize(lookAt-camPos);
    float3 right = normalize(float3(forward.z, 0., -forward.x )); 
    float3 up = cross(forward, right);

    // rd - Unit ray direction.
  //  float3 rd = normalize(forward + FOV*uv.x*right + FOV*uv.y*up);
    
    // Lens distortion, if preferable.
    //float3 rd = (forward + FOV*uv.x*right + FOV*uv.y*up);
    //rd = normalize(float3(rd.xy, rd.z - dot(rd.xy, rd.xy)*.25));    
    
    // Swiveling the camera about the XY-plane (from left to right) when turning corners.
    // Naturally, it's synchronized with the path in some kind of way.
//    rd.xy = mul(rot2( path(lookAt.z).x/16. ),rd.xy);
        
    // Standard ray marching routine. I find that some system setups don't like anything other than
    // a "break" statement (by itself) to exit. 
    float t = trace(camPos, rd);
    
    // Save the object ID just after the "trace" function, since other map calls can change it, which
    // will distort the results.
    saveID = objID; 
    
    // Initialize the scene color.
    float3 sceneCol = float3(0,0,0);
    
    // The ray has effectively hit the surface, so light it up.
    if(t<FAR){
    
    
        // Surface position and surface normal.
        float3 sp = t * rd+camPos;
        float3 sn = getNormal(sp);       

        
        // Function based bump mapping. Comment it out to see the under layer. It's pretty
        // comparable to regular beveled Voronoi... Close enough, anyway.
        if(saveID>.5) sn = doBumpMap(sp, sn, .2);
        else sn = doBumpMap(sp, sn, .008);
        
        // Ambient occlusion.
        float ao = calculateAO(sp, sn);
        
        // Light direction floattors.
        float3 ld = lightPos-sp;

        // Distance from respective lights to the surface point.
        float distlpsp = max(length(ld), 0.001);
        
        // Normalize the light direction floattors.
        ld /= distlpsp;
        
        // Light attenuation, based on the distances above.
        float atten = 1./(1. + distlpsp*0.25); // + distlpsp*distlpsp*0.025
        
        // Ambient light.
        float ambience = 0.5;
        
        // Diffuse lighting.
        float diff = max( dot(sn, ld), 0.0);
    
        // Specular lighting.
        float spec = pow(max( dot( reflect(-ld, sn), -rd ), 0.0 ), 32.);

        
        // Fresnel term. Good for giving a surface a bit of a reflective glow.
        float fre = pow( clamp(dot(sn, rd) + 1., .0, 1.), 1.);
        
     

        // Object texturing and coloring. 
        float3 texCol;        
        
        if(saveID>.5){ // Tunnel walls.
            // Two second texture algorithm. Terrible, but it's dark, so no one will notice. :)
            texCol = float3(.3,.3,.3)*(noise3D(sp*32.)*.66 + noise3D(sp*64.)*.34)*(1.-cellTile(sp*16.)*.75);
            // Darkening the crevices with the bump function. Cheap, but effective.
            texCol *= smoothstep(-.1, .5, cellTile(sp*.75)*.66+cellTile(sp*1.5)*.34)*.85+.15; 
        }
        else { // The biotubes.
            // Cheap, sinewy, vein-like covering. Smoothstepping Voronoi is the main mechanism involved.
            float3 sps = sp/3.;
            float ct = cellTile(sps*2. + sin(sps*12.)*.5)*.66 + cellTile(sps*6. + sin(sps*36.)*.5)*.34;
            texCol = float3(.35, .25, .2)*(1.-smoothstep(-.1, .25, ct)) + float3(0.1, 0.01, 0.004);
        }
        
        
        /////////   
        // Translucency, courtesy of XT95. See the "thickness" function.
        float3 hf =  normalize(ld + sn);
        float th = thickness( sp, sn, 1., 1. );
        float tdiff =  pow( clamp( dot(rd, -hf), 0., 1.), 1.);
        float trans = (tdiff + .0)*th;  
        trans = pow(trans, 4.);        
        ////////        

        
        // Darkening the crevices. Otherwise known as cheap, scientifically-incorrect shadowing.    
        float shading = 1.;//crv*0.5+0.5; 
        
        
        // Shadows - Better, but they really drain the GPU, so I ramped up the fake shadowing so 
        // that it's not as noticeable.
        //shading *= softShadow(sp, ld, 0.05, distlpsp, 8.);
        
        // Combining the above terms to produce the final color. It was based more on acheiving a
        // certain aesthetic than science.
        sceneCol = texCol*(diff + ambience) + float3(.7, .9, 1.)*spec;// + float3(.5, .8, 1)*spec2;
        if(saveID<.5) sceneCol += float3(.7, .9, 1.)*spec*spec;
        sceneCol += texCol*float3(.8, .95, 1)*pow(fre, 4.)*2.;
        sceneCol += float3(1, .07, .15)*trans*1.5;
        
        
        // Fake reflection and refracion on the biotubes. Not a proper reflective and 
        // refracive pass, but it does a reasonable job, and is much cheaper.
        float3 ref, em;
        
        if(saveID<.5){ // Biotubes.
            
            // Fake reflection and refracion to give a bit of a fluid look, albeit
            // in a less than physically correct fashion.
            ref = reflect(rd, sn);
            em = eMap(ref, sn);
            sceneCol += em*.5;
            ref = refract(rd, sn, 1./1.3);//svn*.5 + n*.5
            em = eMap(ref, sn);
            sceneCol += em*float3(2, .2, .3)*1.5;
        }

        // Shading.
        sceneCol *= atten*shading*ao;
       
    
    }
       
    // Blend the scene and the background; It's commented out, but you could also integrate some some 
    // very basic, 8-layered smokey haze.
    //float mist = getMist(camPos, rd, lightPos, t);
    float3 sky = float3(2., .9, .8);//* lerp(1., .75, mist);//*(rd.y*.25 + 1.);
    sceneCol = lerp(sky, sceneCol, 1./(t*t/FAR/FAR*8. + 1.));

    // Clamp and present the pixel to the screen.
    fragColor = float4(sqrt(clamp(sceneCol, 0., 1.)), 1.0);
    


                return fragColor;
            }

            ENDCG
        }
    }
}

// https://www.shadertoy.com/view/4lyGzR
