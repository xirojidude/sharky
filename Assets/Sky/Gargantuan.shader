
Shader "Skybox/Gargantuan"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
        _MainTex2 ("tex2D", 2D) = "white" {}
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
            uniform sampler2D _MainTex2; 

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }

//This is simple render of black hole with gravitational lensing
//Rendering participating media is one of the most complecated part of production rendering
//and rendering heterogenious particifating media illuminated from heterogenious volumetric
//light sorce within gravitational lensing is practically imposible in significant frame rates.
//So I decided to implement only lensing part. lighting isn't calculated.


//random number and cloud generation is taken from iq :)
float seed; //seed initialized in main
float rnd() { return frac(sin(seed++)*43758.5453123); }
//***********************************

//used macros and constants
#define PI                  3.1415926
#define TWO_PI              6.2831852
#define FOUR_PI             12.566370
#define HALF_PI             1.5707963
#define INV_PI              0.3183099
#define INV_TWO_PI          0.1591549
#define INV_FOUR_PI         0.0795775
#define EPSILON             0.00001 
#define IN_RANGE(x,a,b)     (((x) > (a)) && ((x) < (b)))
#define EQUAL_FLT(a,b,eps)  (((a)>((b)-(eps))) && ((a)<((b)+(eps))))
#define IS_ZERO(a)          EQUAL_FLT(a,0.0,EPSILON)

//Increase SPP to remove noise :)
#define SPP 4
#define GRAVITATIONAL_LENSING

struct Ray {
    float3 origin;
    float3 dir;
};

struct Camera {
    float3x3 rotate;
    float3 pos;
    float3 target;
    float fovV;
};

struct BlackHole {
    float3 position_;
    float radius_;
    float ring_radius_inner_;
    float ring_radius_outer_;
    float ring_thickness_;
    float mass_;
};
   
BlackHole gargantua;
Camera camera;

void initScene() {
    gargantua.position_ = float3(0.0, 0.0, -8.0 );
    gargantua.radius_ = 0.1;
    gargantua.ring_radius_inner_ = gargantua.radius_ + 0.8;
    gargantua.ring_radius_outer_ = 6.0;
    gargantua.ring_thickness_ = 0.15;
    gargantua.mass_ = 1000.0;
}

void initCamera( in float3 pos, in float3 target, in float3 upDir, in float fovV ) {
    float3 back = normalize( pos-target );
    float3 right = normalize( cross( upDir, back ) );
    float3 up = cross( back, right );
    camera.rotate[0] = right;
    camera.rotate[1] = up;
    camera.rotate[2] = back;
    camera.fovV = fovV;
    camera.pos = pos;
}

float3 sphericalToCartesian(  in float rho,
                            in float phi,
                            in float theta ) {
    float sinTheta = sin(theta);
    return float3( sinTheta*cos(phi), sinTheta*sin(phi), cos(theta) )*rho;
}

void cartesianToSpherical(  in float3 xyz,
                            out float rho,
                            out float phi,
                            out float theta ) {
    rho = sqrt((xyz.x * xyz.x) + (xyz.y * xyz.y) + (xyz.z * xyz.z));
    phi = asin(xyz.y / rho);
    theta = atan2( xyz.z, xyz.x );
}

Ray genRay( in float2 pixel )
{
    Ray ray;
    
    float2 iPlaneSize=2.*tan(0.5*camera.fovV)*float2(1.,1.);
    float2 ixy=(pixel/float2(800,450) - 0.5)*iPlaneSize;
    
    ray.origin = camera.pos;
    ray.dir = mul(camera.rotate,normalize(float3(ixy.x,ixy.y,-1.0)));

    return ray;
}

float noise( in float3 x ) {
    float3 p = floor(x);
    float3 f = frac(x);
    f = f*f*(3.0-2.0*f);
    float2 uv = ( p.xy + float2(37.0,17.0)*p.z ) + f.xy;
    float2 rg = tex2D( _MainTex, (uv+ 0.5)/256.0 ).yx;              //tex2DLod( _MainTex, (uv+ 0.5)/256.0, 0.0 ).yx;
    return -1.0+2.0*lerp( rg.x, rg.y, f.z );
}

float map5( in float3 p ) {
    float3 q = p;
    float f;
    f  = 0.50000*noise( q ); q = q*2.02;
    f += 0.25000*noise( q ); q = q*2.03;
    f += 0.12500*noise( q ); q = q*2.01;
    f += 0.06250*noise( q ); q = q*2.02;
    f += 0.03125*noise( q );
    return clamp( 1.5 - p.y - 2.0 + 1.75*f, 0.0, 1.0 );
}

//***********************************************************************
// Stars from: nimitz
// https://www.shadertoy.com/view/ltfGDs
//***********************************************************************
float tri(in float x){return abs(frac(x)-.5);}

float3 hash33(float3 p){
    p  = frac(p * float3(5.3983, 5.4427, 6.9371));
    p += dot(p.yzx, p.xyz  + float3(21.5351, 14.3137, 15.3219));
    return frac(float3(p.x * p.z * 95.4337, p.x * p.y * 97.597, p.y * p.z * 93.8365));
}

//smooth and cheap 3d starfield
float3 stars(in float3 p)
{
    float fov = radians(50.0);
    float3 c = float3(0.,0,0);
    float res = (800)*.85*fov;
    
    //Triangular deformation (used to break sphere intersection pattterns)
    p.x += (tri(p.z*50.)+tri(p.y*50.))*0.006;
    p.y += (tri(p.z*50.)+tri(p.x*50.))*0.006;
    p.z += (tri(p.x*50.)+tri(p.y*50.))*0.006;
    
    for (float i=0.;i<3.;i++)
    {
        float3 q = frac(p*(.15*res))-0.5;
        float3 id = floor(p*(.15*res));
        float rn = hash33(id).z;
        float c2 = 1.-smoothstep(-0.2,.4,length(q));
        c2 *= step(rn,0.005+i*0.014);
        c += c2*(lerp(float3(1.0,0.75,0.5),float3(0.85,0.9,1.),rn*30.)*0.5 + 0.5);
        p *= 1.15;
    }
    return c*c*1.5;
}
//*****************************************************************************

float3 getBgColor( float3 dir ) {
    float rho, phi, theta;
    cartesianToSpherical( dir, rho, phi, theta );
    
    float2 uv = float2( phi/PI, theta/TWO_PI );
    float3 c0 = tex2D( _MainTex2, uv).xyz*0.3;
    float3 c1 = stars(dir);
    return c0.bgr*0.4 + c1*2.0;
}
 
void getCloudColorAndDencity(float3 p, float time, out float4 color, out float dencity ) {
    float d2 = dot(p,p);
        
    if( sqrt(d2) < gargantua.radius_ ) {
        dencity = 0.0;
    } else {
        float rho, phi, theta;
        cartesianToSpherical( p, rho, phi, theta );

        //normalize rho
        rho = ( rho - gargantua.ring_radius_inner_)/(gargantua.ring_radius_outer_ - gargantua.ring_radius_inner_);

        if( !IN_RANGE( p.y, -gargantua.ring_thickness_, gargantua.ring_thickness_ ) ||
            !IN_RANGE( rho, 0.0, 1.0 ) ) {
            dencity = 0.0;
        } else {
            float cloudX = sqrt( rho );
            float cloudY = ((p.y - gargantua.position_.y) + gargantua.ring_thickness_ ) / (2.0*gargantua.ring_thickness_);
            float cloudZ = (theta/TWO_PI);

            float blending = 1.0; 

            blending *= lerp(rho*5.0, 1.0 - (rho-0.2)/(0.8*rho), rho>0.2);
            blending *= lerp(cloudY*2.0, 1.0 -(cloudY-0.5)*2.0, cloudY > 0.5);

            float3 moving = float3( time*0.5, 0.0, time*rho*0.1 );

            float3 localCoord = float3( cloudX*(rho*rho), -0.02*cloudY, cloudZ );

            dencity = blending*map5( (localCoord + moving)*100.0 );
            color = 5.0*lerp( float4( 1.0, 0.9, 0.4, rho*dencity ), float4( 1.0, 0.3, 0.1, rho*dencity ), rho );
        }
    }
}

float4 Radiance( in Ray ray )
{
    float4 sum = float4(0.0,0,0,0);

    float marchingStep = lerp( 0.27, 0.3, rnd() );
    float marchingStart = 2.5;
    
    Ray currentRay;  // = Ray ( ray.origin + ray.dir*marchingStart, ray.dir );
    currentRay.origin = ray.origin + ray.dir*marchingStart;
    currentRay.dir = ray.dir;
    
    float transmittance = 1.0;
    
    for(int i=0; i<64 && transmittance > 1e-3; i++) {
        float3 p = currentRay.origin - gargantua.position_;
        
        float dencity;
        float4 ringColor;
        getCloudColorAndDencity(p, _Time.y*0.1, ringColor, dencity);
        
        ringColor *= marchingStep;
        
        float tau = dencity * (1.0 - ringColor.w) * marchingStep;
        transmittance *= exp(-tau);

        sum += transmittance * dencity*ringColor;

#ifdef GRAVITATIONAL_LENSING
        float G_M1_M2 = 0.50;
        float d2 = dot(p,p);
        float3 gravityfloat = normalize(-p)*( G_M1_M2/d2 );
        
        currentRay.dir = normalize( currentRay.dir + marchingStep * gravityfloat );
#endif
        currentRay.origin = currentRay.origin + currentRay.dir*(marchingStep);
    }
    
    float3 bgColor = getBgColor( currentRay.dir );
    sum = float4( bgColor*transmittance + sum.xyz, 1.0 );
    
    return clamp( sum, 0.0, 1.0 );
}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

    seed = _Time.y; ///*_Time.y +*/ // iResolution.y * fragCoord.x / iResolution.x + fragCoord.y / iResolution.y;
    
    initScene();
    
    float2 screen_uv = v.uv; //(iMouse.x!=0.0 && iMouse.y!=0.0)?iMouse.xy/iResolution.xy:float2( 0.8, 0.4 );
    
    float mouseSensitivity = 0.4;
    float3 cameraDir = rd; //sphericalToCartesian( 1.0, -((HALF_PI - (screen_uv.y)*PI)*mouseSensitivity), (-screen_uv.x*TWO_PI)*mouseSensitivity );
    
    initCamera( gargantua.position_ + cameraDir*8.0, gargantua.position_, float3(0.2, 1.0, 0.0), radians(50.0) );
    
    float4 color = float4( 0.0, 0.0, 0.0, 1.0 );
    for( int i=0; i<SPP; i++ ){
        float2 screenCoord = fragCoord.xy + float2( rnd(), rnd() );
        Ray ray = genRay( screenCoord );
        
        color += Radiance( ray );
    }
    
    fragColor = (1.0/float(SPP))*color;

                return fragColor;
            }

            ENDCG
        }
    }
}



